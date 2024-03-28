from typing import List

import vecs
from semantic_router.encoders import BaseEncoder
from tqdm import tqdm

from qdrant_client.http import models as rest
from models.delete import DeleteResponse
from models.document import BaseDocumentChunk
from vectordbs.base import BaseVectorDatabase

MAX_QUERY_TOP_K = 5


class PGVectorService(BaseVectorDatabase):
    def __init__(
        self, index_name: str, dimension: int, credentials: dict, encoder: BaseEncoder
    ):
        super().__init__(
            index_name=index_name,
            dimension=dimension,
            credentials=credentials,
            encoder=encoder,
        )
        client = vecs.create_client(connection_string=credentials["database_uri"])
        self.collection = client.get_or_create_collection(
            name=self.index_name,
            dimension=dimension,
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
        records = []
        for chunk in tqdm(chunks, desc="Upserting to PGVector"):
            records.append(
                (
                    chunk.id,
                    chunk.dense_embedding,
                    {
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "doc_url": chunk.doc_url,
                        **(chunk.metadata if chunk.metadata else {}),
                    },
                )
            )
        self.collection.upsert(records)
        self.collection.create_index()

    async def query(self, input: str, top_k: int = MAX_QUERY_TOP_K) -> List:
        vectors = await self._generate_vectors(input=input)

        results = self.collection.query(
            data=vectors[0],
            limit=top_k,
            include_metadata=True,
            include_value=False,
        )

        chunks = []

        for result in results:
            (
                id,
                metadata,
            ) = result

            chunks.append(
                BaseDocumentChunk(
                    id=id,
                    source_type=metadata.get("filetype"),
                    source=metadata.get("doc_url"),
                    document_id=metadata.get("document_id"),
                    content=metadata.get("content"),
                    doc_url=metadata.get("doc_url"),
                    page_number=metadata.get("page_number"),
                    metadata={**metadata},
                )
            )
        return chunks

    async def delete(self, file_url: str) -> None:
        deleted = self.collection.delete(filters={"doc_url": {"$eq": file_url}})
        return DeleteResponse(num_of_deleted_chunks=len(deleted))
