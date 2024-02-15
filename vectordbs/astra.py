from typing import List

from astrapy.db import AstraDB
from tqdm import tqdm

from semantic_router.encoders import BaseEncoder
from models.document import BaseDocumentChunk
from vectordbs.base import BaseVectorDatabase


class AstraService(BaseVectorDatabase):
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

    async def upsert(self, chunks: List[BaseDocumentChunk]) -> None:
        documents = [
            {
                "_id": chunk.id,
                "text": chunk.content,
                "$vector": chunk.dense_embedding,
                **chunk.metadata,
            }
            for chunk in tqdm(chunks, desc="Upserting to Astra")
        ]
        for i in range(0, len(documents), 5):
            self.collection.insert_many(documents=documents[i : i + 5])

    async def query(self, input: str, top_k: int = 4) -> List:
        vectors = await self._generate_vectors(input=input)
        results = self.collection.vector_find(
            vector=vectors[0],
            limit=top_k,
            fields={"text", "page_number", "source", "document_id"},
        )
        return [
            BaseDocumentChunk(
                id=result.get("_id"),
                document_id=result.get("document_id"),
                content=result.get("text"),
                doc_url=result.get("source"),
                page_number=result.get("page_number"),
            )
            for result in results
        ]

    async def delete(self, file_url: str) -> None:
        self.collection.delete_many(filter={"file_url": file_url})
