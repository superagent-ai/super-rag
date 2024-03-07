import uuid
from typing import List

import weaviate
from semantic_router.encoders import BaseEncoder
from tqdm import tqdm

from models.delete import DeleteResponse
from models.document import BaseDocumentChunk
from utils.logger import logger
from vectordbs.base import BaseVectorDatabase


class WeaviateService(BaseVectorDatabase):
    def __init__(
        self, index_name: str, dimension: int, credentials: dict, encoder: BaseEncoder
    ):
        # TODO: create index if not exists
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

    # TODO: add response model
    async def upsert(self, chunks: List[BaseDocumentChunk]) -> None:
        if not self.client:
            raise ValueError("Weaviate client is not initialized.")

        self.client.batch.configure(
            batch_size=100,
            dynamic=True,
            timeout_retries=3,
            connection_error_retries=3,
            callback=None,
            num_workers=2,
        )

        with self.client.batch as batch:
            for chunk in tqdm(
                chunks, desc=f"Upserting to Weaviate index {self.index_name}"
            ):
                vector_data = {
                    "uuid": chunk.id,
                    "data_object": {
                        "text": chunk.content,
                        "document_id": chunk.document_id,
                        "doc_url": chunk.doc_url,
                        **(chunk.metadata if chunk.metadata else {}),
                    },
                    "class_name": self.index_name,
                    "vector": chunk.dense_embedding,
                }
                batch.add_data_object(**vector_data)
            batch.flush()

    async def query(self, input: str, top_k: int = 25) -> list[BaseDocumentChunk]:
        vectors = await self._generate_vectors(input=input)
        vector = {"vector": vectors[0]}

        try:
            response = (
                self.client.query.get(
                    class_name=self.index_name,
                    properties=["document_id", "text", "doc_url", "page_number"],
                )
                .with_near_vector(vector)
                .with_limit(top_k)
                .do()
            )
            if "data" not in response:
                logger.error(f"Missing 'data' in response: {response}")
                return []

            result_data = response["data"]["Get"][self.index_name]
            document_chunks = []
            for result in result_data:
                document_chunk = BaseDocumentChunk(
                    id=str(uuid.uuid4()),  # TODO: use the actual chunk id from Weaviate
                    document_id=result["document_id"],
                    content=result["text"],
                    doc_url=result["doc_url"],
                    page_number=str(result["page_number"]),
                )

                document_chunks.append(document_chunk)
            return document_chunks
        except KeyError as e:
            logger.error(f"KeyError in response: Missing key {e} - Query: {input}")
            return []
        except Exception as e:
            logger.error(f"Error querying Weaviate: {e}")
            raise Exception(f"Error querying Weaviate: {e}")

    async def delete(self, file_url: str) -> DeleteResponse:
        logger.info(
            f"Deleting from Weaviate index {self.index_name}, file_url: {file_url}"
        )
        result = self.client.batch.delete_objects(
            class_name=self.index_name,
            where={"path": ["doc_url"], "operator": "Equal", "valueText": file_url},
        )
        num_of_deleted_chunks = result.get("results", {}).get("successful", 0)
        return DeleteResponse(num_of_deleted_chunks=num_of_deleted_chunks)
