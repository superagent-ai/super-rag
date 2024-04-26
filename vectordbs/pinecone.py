from typing import List

from pinecone import Pinecone, ServerlessSpec
from semantic_router.encoders import BaseEncoder
from tqdm import tqdm

from models.delete import DeleteResponse
from models.document import BaseDocumentChunk
from models.query import Filter
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
                spec=ServerlessSpec(
                    cloud=credentials["cloud"], region=credentials["region"]
                ),
            )
        self.index = pinecone.Index(name=self.index_name)

    # TODO: add batch size
    async def upsert(self, chunks: List[BaseDocumentChunk], batch_size: int = 100):
        if self.index is None:
            raise ValueError(f"Pinecone index {self.index_name} is not initialized.")
        try:
            # Prepare and upsert documents to Pinecone in batches
            for i in tqdm(range(0, len(chunks), batch_size)):
                i_end = min(i + batch_size, len(chunks))
                chunks_batch = chunks[i:i_end]
                to_upsert = [chunk.to_vector_db() for chunk in chunks_batch]
                self.index.upsert(vectors=to_upsert)
                logger.info(f"Upserted {len(chunks_batch)} chunks into Pinecone")

            # Check that we have all vectors in index
            return self.index.describe_index_stats()
        except Exception as e:
            logger.error(f"Error in embedding documents: {e}")
            raise

    async def query(
        self,
        input: str,
        filter: Filter = None,
        top_k: int = 25,
        include_metadata: bool = True,
    ) -> list[BaseDocumentChunk]:
        if self.index is None:
            raise ValueError(f"Pinecone index {self.index_name} is not initialized.")
        query_vectors = await self._generate_vectors(input=input)
        results = self.index.query(
            vector=query_vectors[0],
            top_k=top_k,
            include_metadata=include_metadata,
            filter=filter,
        )
        chunks = []
        if results.get("matches"):
            for match in results["matches"]:
                chunk = BaseDocumentChunk.from_metadata(metadata=match["metadata"])
                chunks.append(chunk)
            return chunks
        else:
            logger.warning(f"No matches found for the given query '{input}'")
            return []

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
