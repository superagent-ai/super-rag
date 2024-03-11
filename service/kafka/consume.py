import asyncio
from api.ingest import ingest as _ingest, IngestPayload


from service.redis.client import redis_client
from service.redis.ingest_task_manager import IngestTaskManager
from service.kafka.config import ingest_topic
from service.kafka.consumer import get_kafka_consumer

from kafka.consumer.fetcher import ConsumerRecord


async def ingest(msg: ConsumerRecord):
    payload = IngestPayload(**msg.value)
    task_manager = IngestTaskManager(redis_client)
    await _ingest(payload, task_manager)


kafka_actions = {
    ingest_topic: ingest,
}


async def process_msg(msg: ConsumerRecord, topic: str, consumer):
    await kafka_actions[topic](msg)
    consumer.commit()


async def consume():
    consumer = get_kafka_consumer(ingest_topic)

    while True:
        # Response format is {TopicPartiton('topic1', 1): [msg1, msg2]}
        msg_pack = consumer.poll(timeout_ms=3000)

        for tp, messages in msg_pack.items():
            for message in messages:
                await process_msg(message, tp.topic, consumer)


asyncio.run(consume())
