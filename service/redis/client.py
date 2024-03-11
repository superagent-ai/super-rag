from decouple import config
from redis import Redis

redis_client = Redis(
    host=config("REDIS_HOST", "localhost"), port=config("REDIS_PORT", 6379)
)
