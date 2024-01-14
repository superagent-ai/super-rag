from abc import ABC, abstractmethod
from typing import Any, List

import pinecone
from decouple import config
from numpy import ndarray


class VectorDBService(ABC):
    def __init__(self, index_name: str, dimension: int, filter_id: str = None):
        self.index_name = index_name
        self.filter_id = filter_id
        self.dimension = dimension

    @abstractmethod
    def upsert():
        pass

    @abstractmethod
    def query():
        pass

    @abstractmethod
    def rerank(self, query: str, documents: list, top_n: int = 3):
        pass


class PineconeVectorService(VectorDBService):
    def __init__(self, index_name: str, dimension: int, filter_id: str = None):
        super().__init__(
            index_name=index_name, dimension=dimension, filter_id=filter_id
        )
        pinecone.init(
            api_key=config("PINECONE_API_KEY"),
            environment=config("PINECONE_ENVIRONMENT"),
        )
        # Create a new vector index if it doesn't
        # exist dimensions should be passed in the arguments
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name, metric="cosine", shards=1, dimension=dimension
            )
        self.index = pinecone.Index(index_name=self.index_name)

    def upsert(self, vectors: ndarray):
        self.index.upsert(vectors=vectors, namespace=self.filter_id)

    def query(self, queries: List[ndarray], top_k: int, include_metadata: bool = True):
        results = self.index.query(
            queries=queries,
            top_k=top_k,
            include_metadata=include_metadata,
            namespace=self.filter_id,
        )
        return results["results"][0]["matches"]

    def rerank(self, query: str, documents: Any, top_n: int = 3):
        from cohere import Client

        api_key = config("COHERE_API_KEY")
        if not api_key:
            raise ValueError("API key for Cohere is not present.")
        cohere_client = Client(api_key=api_key)
        docs = [
            (
                f"{doc['metadata']['content']}\n\n"
                f"page number: {doc['metadata']['page_label']}"
            )
            for doc in documents
        ]
        re_ranked = cohere_client.rerank(
            model="rerank-multilingual-v2.0",
            query=query,
            documents=docs,
            top_n=top_n,
        ).results
        results = []
        for obj in re_ranked:
            results.append(obj.document["text"])
        return results


def get_vector_service(
    provider: str, index_name: str, filter_id: str = None, dimension: int = 384
):
    services = {
        "PINECONE": PineconeVectorService,
        # Add other providers here
        # e.g "weaviate": WeaviateVectorService,
    }
    service = services.get(provider)
    if service is None:
        raise ValueError(f"Unsupported provider: {provider}")
    return service(index_name=index_name, filter_id=filter_id, dimension=dimension)
