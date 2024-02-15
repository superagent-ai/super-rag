from encoders import BaseEncoder
from encoders.openai import OpenAIEncoder
from models.vector_database import VectorDatabase
from vectordbs.astra import AstraService
from vectordbs.base import BaseVectorDatabase
from vectordbs.pinecone import PineconeService
from vectordbs.qdrant import QdrantService
from vectordbs.weaviate import WeaviateService


def get_vector_service(
    *,
    index_name: str,
    credentials: VectorDatabase,
    encoder: BaseEncoder = OpenAIEncoder(),
) -> BaseVectorDatabase:
    services = {
        "pinecone": PineconeService,
        "qdrant": QdrantService,
        "weaviate": WeaviateService,
        "astra": AstraService,
        # Add other providers here
        # e.g "weaviate": WeaviateVectorService,
    }
    service = services.get(credentials.type.value)
    if service is None:
        raise ValueError(f"Unsupported provider: {credentials.type.value}")
    return service(
        index_name=index_name,
        dimension=encoder.dimension,
        credentials=dict(credentials.config),
        encoder=encoder,
    )
