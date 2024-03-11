from kafka import KafkaProducer

from service.kafka.config import kafka_bootstrap_servers

kafka_producer = KafkaProducer(bootstrap_servers=kafka_bootstrap_servers)
