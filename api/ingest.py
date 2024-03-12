import asyncio
import time
import logging
from typing import Dict

import aiohttp
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from models.api import ApiError
from models.ingest import RequestPayload, TaskStatus
from service.embedding import EmbeddingService
from service.ingest import handle_google_drive, handle_urls
from service.kafka.config import ingest_topic
from service.kafka.producer import kafka_producer
from service.redis.client import redis_client
from service.redis.ingest_task_manager import (
    CreateTaskDto,
    IngestTaskManager,
    UpdateTaskDto,
)
from utils.summarise import SUMMARY_SUFFIX


router = APIRouter()


logger = logging.getLogger(__name__)


class IngestPayload(RequestPayload):
    task_id: str


@router.post("/ingest")
async def add_ingest_queue(payload: RequestPayload):
    try:
        task_manager = IngestTaskManager(redis_client)
        task_id = task_manager.create(CreateTaskDto(status=TaskStatus.PENDING))

        message = IngestPayload(**payload.model_dump(), task_id=str(task_id))

        msg = message.model_dump_json().encode()

        kafka_producer.send(ingest_topic, msg)
        kafka_producer.flush()

        logger.info(f"Task {task_id} added to the queue")

        return {"success": True, "task": {"id": task_id}}

    except Exception as err:
        logger.error(f"Error adding task to the queue: {err}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"sucess": False, "error": {"message": "Internal server error"}},
        )


@router.get("/ingest/tasks/{task_id}")
async def get_task(
    task_id: str,
    long_polling: bool = False,
):
    if long_polling:
        logger.info(f"Long pooling is enabled for task {task_id}")
    else:
        logger.info(f"Long pooling is disabled for task {task_id}")

    task_manager = IngestTaskManager(redis_client)

    def handle_task_not_found(task_id: str):
        logger.warning(f"Task {task_id} not found")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"success": False, "error": {"message": "Task not found"}},
        )

    if not long_polling:
        task = task_manager.get(task_id)
        if not task:
            return handle_task_not_found(task_id)
        return {"success": True, "task": task.model_dump()}
    else:
        start_time = time.time()
        timeout_time = start_time + 30  #  30 seconds from now
        sleep_interval = 3  # seconds

        while start_time < timeout_time:
            task = task_manager.get(task_id)

            if task is None:
                handle_task_not_found(task_id)

            if task.status != TaskStatus.PENDING:
                return {"success": True, "task": task.model_dump()}
            await asyncio.sleep(sleep_interval)

        logger.warning(f"Request timeout for task {task_id}")

        return JSONResponse(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            content={"sucess": False, "error": {"message": "Request timeout"}},
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
        logger.error(f"Error processing ingest task: {e}")
        task_manager.update(
            task_id=payload.task_id,
            task=UpdateTaskDto(
                status=TaskStatus.FAILED,
                error=ApiError(message=str(e)),
            ),
        )
