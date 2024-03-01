from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from semantic_router.encoders import BaseEncoder
from tqdm import tqdm

from models.delete import DeleteResponse
from models.document import BaseDocumentChunk
from vectordbs.base import BaseVectorDatabase

MAX_QUERY_TOP_K = 5


class QdrantService(BaseVectorDatabase):
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

    async def upsert(self, chunks: List[BaseDocumentChunk]) -> None:
        points = []
        for chunk in tqdm(chunks, desc="Upserting to Qdrant"):
            points.append(
                rest.PointStruct(
                    id=chunk.id,
                    vector={"content": chunk.dense_embedding},
                    payload={
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "doc_url": chunk.doc_url,
                        **(chunk.metadata if chunk.metadata else {}),
                    },
                )
            )

        self.client.upsert(collection_name=self.index_name, wait=True, points=points)

    async def query(self, input: str, top_k: int = MAX_QUERY_TOP_K) -> List:
        vectors = await self._generate_vectors(input=input)
        search_result = self.client.search(
            collection_name=self.index_name,
            query_vector=("content", vectors[0]),
            limit=top_k,
            with_payload=True,
        )
        return [
            BaseDocumentChunk(
                id=result.id,
                document_id=result.payload.get("document_id"),
                content=result.payload.get("content"),
                doc_url=result.payload.get("doc_url"),
                page_number=result.payload.get("page_number"),
                metadata={**result.payload},
            )
            for result in search_result
        ]

    async def delete(self, file_url: str) -> None:

        #         client.count(
        #     collection_name="{collection_name}",
        #     count_filter=models.Filter(
        #         must=[
        #             models.FieldCondition(key="color", match=models.MatchValue(value="red")),
        #         ]
        #     ),
        #     exact=True,
        # )

        deleted_chunks = self.client.count(
            collection_name=self.index_name,
            count_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="file_url", match=rest.MatchValue(value=file_url)
                    )
                ]
            ),
            exact=True,
        )

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

        return DeleteResponse(num_of_deleted_chunks=deleted_chunks.count)
