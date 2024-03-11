from redis import Redis
from decouple import config

redis_client = Redis(
    host=config("REDIS_HOST", "localhost"), port=config("REDIS_PORT", 6379)
)
