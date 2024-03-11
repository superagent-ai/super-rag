import json

from redis import Redis

from models.ingest import IngestTaskResponse


class CreateTaskDto(IngestTaskResponse):
    pass


class UpdateTaskDto(IngestTaskResponse):
    pass


class IngestTaskManager:
    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client

    def create(self, task: CreateTaskDto):
        task_id = self.redis_client.incr("task_id")
        self.redis_client.set(task_id, task.model_dump_json())
        return task_id

    def get(self, task_id):
        return IngestTaskResponse(**json.loads(self.redis_client.get(task_id)))

    def update(self, task_id, task: UpdateTaskDto):
        self.redis_client.set(task_id, task.model_dump_json())

    def delete(self, task_id):
        self.redis_client.delete(task_id)
