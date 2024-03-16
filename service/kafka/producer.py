from kafka import KafkaProducer
from decouple import config
from service.kafka.config import kafka_bootstrap_servers

kafka_producer = KafkaProducer(
    bootstrap_servers=kafka_bootstrap_servers,
    security_protocol=config("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
    sasl_mechanism=config("KAFKA_SASL_MECHANISM", "PLAIN"),
    sasl_plain_username=config("KAFKA_SASL_PLAIN_USERNAME", None),
    sasl_plain_password=config("KAFKA_SASL_PLAIN_PASSWORD", None),
    api_version_auto_timeout_ms=100000,
)
