import json

from redis import Redis

from models.ingest import IngestTaskResponse


class CreateTaskDto(IngestTaskResponse):
    pass


class UpdateTaskDto(IngestTaskResponse):
    pass


class IngestTaskManager:
    TASK_PREFIX = "ingest:task:"
    INGESTION_TASK_ID_KEY = "ingestion_task_id"

    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client

    def _get_task_key(self, task_id):
        return f"{self.TASK_PREFIX}{task_id}"

    def create(self, task: CreateTaskDto):
        task_id = self.redis_client.incr(self.INGESTION_TASK_ID_KEY)
        task_key = self._get_task_key(task_id)
        self.redis_client.set(task_key, task.model_dump_json())
        return task_id

    def get(self, task_id):
        task_key = self._get_task_key(task_id)
        task_data = self.redis_client.get(task_key)

        if task_data:
            return IngestTaskResponse(**json.loads(task_data))
        else:
            return None

    def update(self, task_id, task: UpdateTaskDto):
        task_key = self._get_task_key(task_id)
        self.redis_client.set(task_key, task.model_dump_json())

    def delete(self, task_id):
        task_key = self._get_task_key(task_id)
        self.redis_client.delete(task_key)
