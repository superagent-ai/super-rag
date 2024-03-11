from decouple import config

ingest_topic = config("KAFKA_TOPIC_INGEST", default="ingestion")


kafka_bootstrap_servers: str = config(
    "KAFKA_BOOTSTRAP_SERVERS", default="localhost:9092"
)
