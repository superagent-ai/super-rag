from abc import ABC, abstractmethod
from typing import Any, List

import weaviate
from astrapy.db import AstraDB
from decouple import config
from pinecone import Pinecone, ServerlessSpec
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from tqdm import tqdm

from encoders.base import BaseEncoder
from encoders.openai import OpenAIEncoder
from models.document import BaseDocumentChunk
from models.vector_database import VectorDatabase
from utils.logger import logger


class VectorService(ABC):
    def __init__(
        self, index_name: str, dimension: int, credentials: dict, encoder: BaseEncoder
    ):
        self.index_name = index_name
        self.dimension = dimension
        self.credentials = credentials
        self.encoder = encoder

    @abstractmethod
    async def upsert():
        pass

    @abstractmethod
    async def query(self, input: str, top_k: int = 25) -> List[BaseDocumentChunk]:
        pass

    # @abstractmethod
    # async def convert_to_rerank_format():
    #     pass

    @abstractmethod
    async def delete(self, file_url: str):
        pass

    async def _generate_vectors(self, input: str) -> List[List[float]]:
        return self.encoder([input])

    async def rerank(
        self, query: str, documents: list[BaseDocumentChunk], top_n: int = 5
    ) -> list[BaseDocumentChunk]:
        from cohere import Client

        api_key = config("COHERE_API_KEY")
        if not api_key:
            raise ValueError("API key for Cohere is not present.")
        cohere_client = Client(api_key=api_key)

        # Avoid duplications, TODO: fix ingestion for duplications
        # Deduplicate documents based on content while preserving order
        seen = set()
        deduplicated_documents = [
            doc
            for doc in documents
            if doc.content not in seen and not seen.add(doc.content)
        ]
        docs_text = list(
            doc.content
            for doc in tqdm(
                deduplicated_documents,
                desc=f"Reranking {len(deduplicated_documents)} documents",
            )
        )
        try:
            re_ranked = cohere_client.rerank(
                model="rerank-multilingual-v2.0",
                query=query,
                documents=docs_text,
                top_n=top_n,
            ).results
            results = []
            for r in tqdm(re_ranked, desc="Processing reranked results "):
                doc = deduplicated_documents[r.index]
                results.append(doc)
            return results
        except Exception as e:
            logger.error(f"Error while reranking: {e}")
            raise Exception(f"Error while reranking: {e}")


class PineconeVectorService(VectorService):
    def __init__(
        self, index_name: str, dimension: int, credentials: dict, encoder: BaseEncoder
    ):
        super().__init__(
            index_name=index_name,
            dimension=dimension,
            credentials=credentials,
            encoder=encoder,
        )
        pinecone = Pinecone(api_key=credentials["api_key"])
        if index_name not in [index.name for index in pinecone.list_indexes()]:
            pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-west-2"),
            )
        self.index = pinecone.Index(name=self.index_name)

    async def upsert(self, embeddings: List[tuple[str, list, dict[str, Any]]]):
        if self.index is None:
            raise ValueError(f"Pinecone index {self.index_name} is not initialized.")
        for _ in tqdm(
            embeddings, desc=f"Upserting to Pinecone index {self.index_name}"
        ):
            pass
        self.index.upsert(vectors=embeddings)

    async def query(
        self, input: str, top_k: int = 25, include_metadata: bool = True
    ) -> list[BaseDocumentChunk]:
        if self.index is None:
            raise ValueError(f"Pinecone index {self.index_name} is not initialized.")
        query_vectors = await self._generate_vectors(input=input)
        results = self.index.query(
            vector=query_vectors[0],
            top_k=top_k,
            include_metadata=include_metadata,
        )
        document_chunks = []
        for match in results["matches"]:
            document_chunk = BaseDocumentChunk(
                id=match["id"],
                document_id=match["metadata"].get("document_id", ""),
                content=match["metadata"]["content"],
                doc_url=match["metadata"].get("source", ""),
                page_number=str(
                    match["metadata"].get("page_number", "")
                ),  # TODO: is this correct?
                metadata={
                    key: value
                    for key, value in match["metadata"].items()
                    if key not in ["content", "file_url"]
                },
            )
            document_chunks.append(document_chunk)
        return document_chunks

    async def delete(self, file_url: str) -> dict[str, int]:
        if self.index is None:
            raise ValueError(f"Pinecone index {self.index_name} is not initialized.")

        query_response = self.index.query(
            vector=[0.0] * self.dimension,
            top_k=1000,
            include_metadata=True,
            filter={"source": {"$eq": file_url}},
        )
        chunks = query_response.matches
        logger.info(
            f"Deleting {len(chunks)} chunks from Pinecone {self.index_name} index."
        )

        if chunks:
            self.index.delete(ids=[chunk["id"] for chunk in chunks])
        return {"num_of_deleted_chunks": len(chunks)}


class QdrantService(VectorService):
    def __init__(
        self, index_name: str, dimension: int, credentials: dict, encoder: BaseEncoder
    ):
        super().__init__(
            index_name=index_name,
            dimension=dimension,
            credentials=credentials,
            encoder=encoder,
        )
        self.client = QdrantClient(
            url=credentials["host"], api_key=credentials["api_key"], https=True
        )
        collections = self.client.get_collections()
        if index_name not in [c.name for c in collections.collections]:
            self.client.create_collection(
                collection_name=self.index_name,
                vectors_config={
                    "content": rest.VectorParams(
                        size=dimension, distance=rest.Distance.COSINE
                    )
                },
                optimizers_config=rest.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
            )

    # TODO: remove this
    async def convert_to_rerank_format(self, chunks: List[rest.PointStruct]):
        docs = [
            {
                "content": chunk.payload.get("content"),
                "page_label": chunk.payload.get("page_label"),
                "file_url": chunk.payload.get("file_url"),
            }
            for chunk in chunks
        ]
        return docs

    async def upsert(self, embeddings: List[tuple[str, list, dict[str, Any]]]) -> None:
        points = []
        for _embedding in tqdm(embeddings, desc="Upserting to Qdrant"):
            points.append(
                rest.PointStruct(
                    id=_embedding[0],
                    vector={"content": _embedding[1]},
                    payload={**_embedding[2]},
                )
            )

        self.client.upsert(collection_name=self.index_name, wait=True, points=points)

    async def query(self, input: str, top_k: int) -> List:
        vectors = await self._generate_vectors(input=input)
        search_result = self.client.search(
            collection_name=self.index_name,
            query_vector=("content", vectors),
            limit=top_k,
            # query_filter=rest.Filter(
            #    must=[
            #        rest.FieldCondition(
            #            key="datasource_id",
            #            match=rest.MatchValue(value=datasource_id),
            #        ),
            #    ]
            # ),
            with_payload=True,
        )
        # TODO: return list[BaseDocumentChunk]
        return search_result

    async def delete(self, file_url: str) -> None:
        self.client.delete(
            collection_name=self.index_name,
            points_selector=rest.FilterSelector(
                filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="file_url", match=rest.MatchValue(value=file_url)
                        )
                    ]
                )
            ),
        )


class WeaviateService(VectorService):
    def __init__(
        self, index_name: str, dimension: int, credentials: dict, encoder: BaseEncoder
    ):
        super().__init__(
            index_name=index_name,
            dimension=dimension,
            credentials=credentials,
            encoder=encoder,
        )
        self.client = weaviate.Client(
            url=credentials["host"],
            auth_client_secret=weaviate.AuthApiKey(api_key=credentials["api_key"]),
        )
        schema = {
            "class": self.index_name,
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                }
            ],
        }
        if not self.client.schema.exists(self.index_name):
            self.client.schema.create_class(schema)

    # TODO: remove this
    async def convert_to_rerank_format(self, chunks: List) -> List:
        docs = [
            {
                "content": chunk.get("text"),
                "page_label": chunk.get("page_label"),
                "file_url": chunk.get("file_url"),
            }
            for chunk in chunks
        ]
        return docs

    async def upsert(self, embeddings: List[tuple[str, list, dict[str, Any]]]) -> None:
        with self.client.batch as batch:
            for _embedding in tqdm(embeddings, desc="Upserting to Weaviate"):
                params = {
                    "uuid": _embedding[0],
                    "data_object": {"text": _embedding[2]["content"], **_embedding[2]},
                    "class_name": self.index_name,
                    "vector": _embedding[1],
                }
                batch.add_data_object(**params)
            batch.flush()

    async def query(self, input: str, top_k: int = 4) -> List:
        vectors = await self._generate_vectors(input=input)
        vector = {"vector": vectors}
        result = (
            self.client.query.get(
                self.index_name.capitalize(),
                ["text", "file_url", "page_label"],
            )
            .with_near_vector(vector)
            .with_limit(top_k)
            .do()
        )
        # TODO: return list[BaseDocumentChunk]
        return result["data"]["Get"][self.index_name.capitalize()]

    async def delete(self, file_url: str) -> None:
        self.client.batch.delete_objects(
            class_name=self.index_name,
            where={"path": ["file_url"], "operator": "Equal", "valueText": file_url},
        )


class AstraService(VectorService):
    def __init__(
        self, index_name: str, dimension: int, credentials: dict, encoder: BaseEncoder
    ):
        super().__init__(
            index_name=index_name,
            dimension=dimension,
            credentials=credentials,
            encoder=encoder,
        )
        self.client = AstraDB(
            token=credentials["api_key"],
            api_endpoint=credentials["host"],
        )
        collections = self.client.get_collections()
        if self.index_name not in collections["status"]["collections"]:
            self.collection = self.client.create_collection(
                dimension=dimension, collection_name=index_name
            )
        self.collection = self.client.collection(collection_name=self.index_name)

    # TODO: remove this
    async def convert_to_rerank_format(self, chunks: List) -> List:
        docs = [
            {
                "content": chunk.get("text"),
                "page_label": chunk.get("page_label"),
                "file_url": chunk.get("file_url"),
            }
            for chunk in chunks
        ]
        return docs

    async def upsert(self, embeddings: List[tuple[str, list, dict[str, Any]]]) -> None:
        documents = [
            {
                "_id": _embedding[0],
                "text": _embedding[2]["content"],
                "$vector": _embedding[1],
                **_embedding[2],
            }
            for _embedding in tqdm(embeddings, desc="Upserting to Astra")
        ]
        for i in range(0, len(documents), 5):
            self.collection.insert_many(documents=documents[i : i + 5])

    async def query(self, input: str, top_k: int = 4) -> List:
        vectors = await self._generate_vectors(input=input)
        results = self.collection.vector_find(
            vector=vectors, limit=top_k, fields={"text", "page_label", "file_url"}
        )
        # TODO: return list[BaseDocumentChunk]
        return results

    async def delete(self, file_url: str) -> None:
        self.collection.delete_many(filter={"file_url": file_url})


def get_vector_service(
    *,
    index_name: str,
    credentials: VectorDatabase,
    encoder: BaseEncoder = OpenAIEncoder(),
) -> VectorService:
    services = {
        "pinecone": PineconeVectorService,
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
