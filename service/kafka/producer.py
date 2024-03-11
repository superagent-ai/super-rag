from service.kafka.config import kafka_bootstrap_servers
from kafka import KafkaProducer

kafka_producer = KafkaProducer(bootstrap_servers=kafka_bootstrap_servers)
