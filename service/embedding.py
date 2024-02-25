import asyncio
import copy
import uuid
from tempfile import NamedTemporaryFile
from typing import Any, List, Literal, Optional

import numpy as np
import requests
import tiktoken
from decouple import config
from semantic_router.encoders import (
    BaseEncoder,
    CohereEncoder,
    OpenAIEncoder,
)
from tqdm import tqdm
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

from models.document import BaseDocument, BaseDocumentChunk
from models.file import File
from models.google_drive import GoogleDrive
from models.ingest import ChunkConfig, Encoder, EncoderEnum
from service.splitter import UnstructuredSemanticSplitter
from utils.logger import logger
from utils.summarise import completion
from vectordbs import get_vector_service

# TODO: Add similarity score to the BaseDocumentChunk
# TODO: Add relevance score to the BaseDocumentChunk
# TODO: Add created_at date to the BaseDocumentChunk


class EmbeddingService:
    def __init__(
        self,
        index_name: str,
        encoder: BaseEncoder,
        vector_credentials: dict,
        dimensions: Optional[int],
        files: Optional[List[File]] = None,
        google_drive: Optional[GoogleDrive] = None,
    ):
        self.encoder = encoder
        self.files = files
        self.google_drive = google_drive
        self.index_name = index_name
        self.vector_credentials = vector_credentials
        self.dimensions = dimensions
        self.unstructured_client = UnstructuredClient(
            api_key_auth=config("UNSTRUCTURED_IO_API_KEY"),
            server_url=config("UNSTRUCTURED_IO_SERVER_URL"),
        )

    def _get_strategy(self, type: str) -> Optional[str]:
        strategies = {
            "PDF": "auto",
        }
        try:
            return strategies[type]
        except KeyError:
            return None

    async def _partition_file(
        self,
        file: File,
        strategy="auto",
        returned_elements_type: Literal["chunked", "original"] = "chunked",
    ) -> List[Any]:
        """
        Downloads the file and extracts elements using the partition function.
        Returns a list of unstructured elements.
        """
        # TODO: This will overwrite the function parameter?
        # if file.type is None:
        #     raise ValueError(f"File type not set for {file.url}")
        # strategy = self._get_strategy(type=file.type.value)

        # TODO: Do we need this if we have default value in the function signature?
        # if strategy is None:
        #     strategy = "auto"

        logger.info(
            f"Downloading and extracting elements from {file.url},"
            f"using `{strategy}` strategy"
        )
        with NamedTemporaryFile(suffix=file.suffix, delete=True) as temp_file:
            with requests.get(url=file.url) as response:
                temp_file.write(response.content)
                temp_file.flush()
                temp_file.seek(0)  # Reset file pointer to the beginning
                file_content = temp_file.read()
                file_name = temp_file.name

            files = shared.Files(
                content=file_content,
                file_name=file_name,
            )
            if returned_elements_type == "original":
                req = shared.PartitionParameters(
                    files=files,
                    include_page_breaks=True,
                    strategy=strategy,
                )
            else:
                req = shared.PartitionParameters(
                    files=files,
                    include_page_breaks=True,
                    strategy=strategy,
                    max_characters=2500,
                    new_after_n_chars=1000,
                    chunking_strategy="by_title",
                )
            try:
                unstructured_response = self.unstructured_client.general.partition(req)
                if unstructured_response.elements is not None:
                    return unstructured_response.elements
                else:
                    logger.error(
                        f"Error partitioning file {file.url}: {unstructured_response}"
                    )
                    return []
            except SDKError as e:
                logger.error(f"Error partitioning file {file.url}: {e}")
                return []

    def _tiktoken_length(self, text: str):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text, disallowed_special=())
        return len(tokens)

    def _sanitize_metadata(self, metadata: dict) -> dict:
        def sanitize_value(value):
            if isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, list):
                # Ensure all elements in the list are of type str, int, float, or bool
                # Convert non-compliant elements to str
                sanitized_list = []
                for v in value:
                    if isinstance(v, (str, int, float, bool)):
                        sanitized_list.append(v)
                    elif isinstance(v, (dict, list)):
                        # For nested structures, convert to a string representation
                        sanitized_list.append(str(v))
                    else:
                        sanitized_list.append(str(v))
                return sanitized_list
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            else:
                return str(value)

        return {key: sanitize_value(value) for key, value in metadata.items()}

    async def generate_chunks(
        self,
        config: ChunkConfig,
    ) -> List[BaseDocumentChunk]:
        doc_chunks = []
        for file in tqdm(self.files, desc="Generating chunks"):
            try:
                chunks = []
                if config.split_method == "by_title":
                    chunked_elements = await self._partition_file(
                        file, strategy=config.partition_strategy
                    )
                    # TODO: handle chunked_elements being None
                    for element in chunked_elements:
                        chunk_data = {
                            "content": element.get("text"),
                            "metadata": self._sanitize_metadata(
                                element.get("metadata")
                            ),
                        }
                        chunks.append(chunk_data)

                if config.split_method == "semantic":
                    elements = await self._partition_file(
                        file,
                        strategy=config.partition_strategy,
                        returned_elements_type="original",
                    )
                    splitter = UnstructuredSemanticSplitter(
                        encoder=self.encoder,
                        window_size=config.rolling_window_size,
                        min_split_tokens=config.min_chunk_tokens,
                        max_split_tokens=config.max_token_size,
                    )
                    chunks = await splitter(elements=elements)

                if not chunks:
                    continue

                doc_content = " ".join(
                    [str(chunk.get("content", "")) for chunk in chunks]
                )
                document = BaseDocument(
                    id=f"doc_{uuid.uuid4()}",
                    content=doc_content,
                    doc_url=file.url,
                    metadata={
                        "source": file.url,
                        "source_type": "document",
                        "document_type": file.suffix,
                    },
                )

                for chunk in chunks:
                    chunk_id = str(uuid.uuid4())
                    chunk_with_title = (
                        f"{chunk.get('title', '')}\n{chunk.get('content', '')}"
                    )
                    doc_chunk = BaseDocumentChunk(
                        id=chunk_id,
                        doc_url=file.url,
                        document_id=document.id,
                        content=chunk_with_title,
                        source=file.url,
                        source_type=file.suffix,
                        chunk_index=chunk.get("chunk_index", None),
                        title=chunk.get("title", None),
                        token_count=self._tiktoken_length(chunk.get("content", "")),
                        metadata=self._sanitize_metadata(chunk.get("metadata", {})),
                    )
                    doc_chunks.append(doc_chunk)
            except Exception as e:
                logger.error(f"Error loading chunks from {file.url}: {e}")
                raise
        return doc_chunks

    async def embed_and_upsert(
        self,
        chunks: List[BaseDocumentChunk],
        encoder: BaseEncoder,
        index_name: Optional[str] = None,
        batch_size: int = 100,
    ) -> List[BaseDocumentChunk]:
        pbar = tqdm(total=len(chunks), desc="Generating embeddings")
        sem = asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

        async def embed_batch(
            chunks_batch: List[BaseDocumentChunk],
        ) -> List[BaseDocumentChunk]:
            async with sem:
                try:
                    texts = [chunk.content for chunk in chunks_batch]
                    embeddings = encoder(texts)
                    for chunk, embedding in zip(chunks_batch, embeddings):
                        chunk.dense_embedding = np.array(embedding).tolist()
                    return chunks_batch
                except Exception as e:
                    logger.error(f"Error embedding a batch of documents: {e}")
                    raise

        # Create batches of chunks
        chunks_batches = [
            chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)
        ]

        # Process each batch
        tasks = [embed_batch(batch) for batch in chunks_batches]
        chunks_with_embeddings = await asyncio.gather(*tasks)
        chunks_with_embeddings = [
            chunk
            for batch in chunks_with_embeddings
            for chunk in batch
            if chunk is not None
        ]
        pbar.close()

        vector_service = get_vector_service(
            index_name=index_name or self.index_name,
            credentials=self.vector_credentials,
            encoder=encoder,
            dimensions=self.dimensions,
        )
        try:
            await vector_service.upsert(chunks=chunks_with_embeddings)
        except Exception as e:
            logger.error(f"Error upserting embeddings: {e}")
            raise

        return chunks_with_embeddings

    async def generate_summary_documents(
        self, documents: List[BaseDocumentChunk]
    ) -> List[BaseDocumentChunk]:
        pbar = tqdm(total=len(documents), desc="Grouping chunks")
        pages = {}
        for document in documents:
            page_number = document.metadata.get("page_number", None)
            if page_number not in pages:
                pages[page_number] = copy.deepcopy(document)
            else:
                pages[page_number].content += document.content
            pbar.update()
        pbar.close()

        # Limit to 10 concurrent jobs
        sem = asyncio.Semaphore(10)

        async def safe_completion(document: BaseDocumentChunk) -> BaseDocumentChunk:
            async with sem:
                try:
                    document.content = await completion(document=document)
                    pbar.update()
                    return document
                except Exception as e:
                    logger.error(f"Error summarizing document {document.id}: {e}")
                    return None

        pbar = tqdm(total=len(pages), desc="Summarizing documents")
        tasks = [safe_completion(document) for document in pages.values()]
        summary_documents = await asyncio.gather(*tasks, return_exceptions=False)
        pbar.close()

        return summary_documents


def get_encoder(*, encoder_config: Encoder) -> BaseEncoder:
    encoder_mapping = {
        EncoderEnum.cohere: CohereEncoder,
        EncoderEnum.openai: OpenAIEncoder,
    }
    encoder_provider = encoder_config.provider

    encoder = encoder_config.name
    encoder_class = encoder_mapping.get(encoder_provider)
    if encoder_class is None:
        raise ValueError(f"Unsupported provider: {encoder_provider}")
    return encoder_class(name=encoder)
