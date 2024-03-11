from kafka import KafkaConsumer
from service.kafka.config import kafka_bootstrap_servers
import json


def get_kafka_consumer(topic: str):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=kafka_bootstrap_servers,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode("ascii")),
        group_id="my-group",
    )

    return consumer
