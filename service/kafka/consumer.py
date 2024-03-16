import json

from kafka import KafkaConsumer
from decouple import config

from service.kafka.config import kafka_bootstrap_servers


def get_kafka_consumer(topic: str):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=kafka_bootstrap_servers,
        group_id="my-group",
        security_protocol=config("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
        sasl_mechanism=config("KAFKA_SASL_MECHANISM", "PLAIN"),
        sasl_plain_username=config("KAFKA_SASL_PLAIN_USERNAME", None),
        sasl_plain_password=config("KAFKA_SASL_PLAIN_PASSWORD", None),
        auto_offset_reset="earliest",
        value_deserializer=lambda m: json.loads(m.decode("ascii")),
        enable_auto_commit=False,
    )

    return consumer
