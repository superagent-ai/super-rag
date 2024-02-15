from abc import ABC, abstractmethod
from typing import List

from decouple import config
from semantic_router.encoders import BaseEncoder
from tqdm import tqdm

from models.delete import DeleteResponse
from models.document import BaseDocumentChunk
from utils.logger import logger


class BaseVectorDatabase(ABC):
    def __init__(
        self, index_name: str, dimension: int, credentials: dict, encoder: BaseEncoder
    ):
        self.index_name = index_name
        self.dimension = dimension
        self.credentials = credentials
        self.encoder = encoder

    @abstractmethod
    async def upsert(self, chunks: List[BaseDocumentChunk]):
        pass

    @abstractmethod
    async def query(self, input: str, top_k: int = 25) -> List[BaseDocumentChunk]:
        pass

    @abstractmethod
    async def delete(self, file_url: str) -> DeleteResponse:
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
