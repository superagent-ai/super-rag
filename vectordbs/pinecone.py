from typing import List

from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from semantic_router.encoders import BaseEncoder
from models.delete import DeleteResponse
from models.document import BaseDocumentChunk
from utils.logger import logger
from vectordbs.base import BaseVectorDatabase


class PineconeService(BaseVectorDatabase):
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

    # TODO: add batch size
    async def upsert(self, chunks: List[BaseDocumentChunk]):
        if self.index is None:
            raise ValueError(f"Pinecone index {self.index_name} is not initialized.")

        # Prepare the data for upserting into Pinecone
        vectors_to_upsert = []
        for chunk in tqdm(
            chunks,
            desc=f"Upserting {len(chunks)} chunks to Pinecone index {self.index_name}",
        ):
            vector_data = {
                "id": chunk.id,
                "values": chunk.dense_embedding,
                "metadata": {
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "doc_url": chunk.doc_url,
                    **(chunk.metadata if chunk.metadata else {}),
                },
            }
            vectors_to_upsert.append(vector_data)
        self.index.upsert(vectors=vectors_to_upsert)

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

    async def delete(self, file_url: str) -> DeleteResponse:
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
        return DeleteResponse(num_of_deleted_chunks=len(chunks))
