import asyncio
import copy
import uuid
import mimetypes
from tempfile import NamedTemporaryFile
from typing import Any, List, Optional

import numpy as np
import requests
from semantic_router.encoders import (
    BaseEncoder,
    CohereEncoder,
    HuggingFaceEncoder,
    OpenAIEncoder,
)
from tqdm import tqdm
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition

from models.document import BaseDocument, BaseDocumentChunk
from models.file import File
from models.ingest import Encoder, EncoderEnum
from utils.logger import logger
from utils.summarise import completion
from vectordbs import get_vector_service


class EmbeddingService:
    def __init__(
        self,
        files: List[File],
        index_name: str,
        vector_credentials: dict,
        dimensions: Optional[int],
    ):
        self.files = files
        self.index_name = index_name
        self.vector_credentials = vector_credentials
        self.dimensions = dimensions

    def _get_datasource_suffix(self, type: str) -> dict:
        suffixes = {
            "TXT": ".txt",
            "PDF": ".pdf",
            "MARKDOWN": ".md",
            "DOCX": ".docx",
            "CSV": ".csv",
            "XLSX": ".xlsx",
        }
        try:
            return suffixes[type]
        except KeyError:
            raise ValueError("Unsupported datasource type")

    def _get_strategy(self, type: str) -> dict:
        strategies = {
            "PDF": "auto",
        }
        try:
            return strategies[type]
        except KeyError:
            return None

    async def _download_and_extract_elements(
        self, file, strategy: Optional[str] = "hi_res"
    ) -> List[Element]:
        """
        Downloads the file and extracts elements using the partition function.
        Returns a list of unstructured elements.
        """
        logger.info(
            f"Downloading and extracting elements from {file.url}, "
            f"using `{strategy}` strategy"
        )
        suffix = self._get_datasource_suffix(file.type.value)
        strategy = self._get_strategy(type=file.type.value)
        with NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
            with requests.get(url=file.url) as response:
                print(response)
                temp_file.write(response.content)
                temp_file.flush()
            content_type = mimetypes.guess_type(temp_file.name)[0]
            print(content_type)
            elements = partition(
                file=temp_file,
                include_page_breaks=True,
                strategy=strategy,
                content_type=content_type,
            )
        return elements

    async def generate_document(
        self, file: File, elements: List[Any]
    ) -> BaseDocument | None:
        logger.info(f"Generating document from {file.url}")
        try:
            doc_content = "".join(element.text for element in elements)
            if not doc_content:
                logger.error(f"Cannot extract text from {file.url}")
                return None
            doc_metadata = {
                "source": file.url,
                "source_type": "document",
                "document_type": self._get_datasource_suffix(file.type.value),
            }
            return BaseDocument(
                id=f"doc_{uuid.uuid4()}",
                content=doc_content,
                doc_url=file.url,
                metadata=doc_metadata,
            )
        except Exception as e:
            logger.error(f"Error loading document {file.url}: {e}")

    async def generate_chunks(
        self, strategy: Optional[str] = "auto"
    ) -> List[BaseDocumentChunk]:
        doc_chunks = []
        for file in tqdm(self.files, desc="Generating chunks"):
            try:
                elements = await self._download_and_extract_elements(file, strategy)
                document = await self.generate_document(file, elements)
                if not document:
                    continue
                chunks = chunk_by_title(
                    elements, max_characters=1500, new_after_n_chars=1000
                )
                for chunk in chunks:
                    # Ensure all metadata values are of a type acceptable
                    sanitized_metadata = {
                        key: (
                            value
                            if isinstance(value, (str, int, float, bool, list))
                            else str(value)
                        )
                        for key, value in chunk.metadata.to_dict().items()
                    }
                    chunk_id = str(uuid.uuid4())  # must be a valid UUID
                    doc_chunks.append(
                        BaseDocumentChunk(
                            id=chunk_id,
                            document_id=document.id,
                            content=chunk.text,
                            doc_url=file.url,
                            metadata={
                                "chunk_id": chunk_id,
                                "document_id": document.id,
                                "source": file.url,
                                "source_type": "document",
                                "document_type": self._get_datasource_suffix(
                                    file.type.value
                                ),
                                "content": chunk.text,
                                **sanitized_metadata,
                            },
                        )
                    )
            except Exception as e:
                logger.error(f"Error loading chunks from {file.url}: {e}")
        return doc_chunks

    async def generate_and_upsert_embeddings(
        self,
        documents: List[BaseDocumentChunk],
        encoder: BaseEncoder,
        index_name: Optional[str] = None,
    ) -> List[BaseDocumentChunk]:
        pbar = tqdm(total=len(documents), desc="Generating embeddings")
        sem = asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

        async def safe_generate_embedding(
            chunk: BaseDocumentChunk,
        ) -> BaseDocumentChunk | None:
            async with sem:
                try:
                    return await generate_embedding(chunk)
                except Exception as e:
                    logger.error(f"Error embedding document {chunk.id}: {e}")
                    return None

        async def generate_embedding(
            chunk: BaseDocumentChunk,
        ) -> BaseDocumentChunk | None:
            if chunk is not None:
                embeddings: List[np.ndarray] = [
                    np.array(e) for e in encoder([chunk.content])
                ]
                chunk.dense_embedding = embeddings[0].tolist()
                pbar.update()
                return chunk

        tasks = [safe_generate_embedding(document) for document in documents]
        chunks_with_embeddings = await asyncio.gather(*tasks, return_exceptions=False)
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
            raise Exception(f"Error upserting embeddings: {e}")

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
        EncoderEnum.huggingface: HuggingFaceEncoder,
    }
    encoder_provider = encoder_config.type
    encoder = encoder_config.name
    encoder_class = encoder_mapping.get(encoder_provider)
    if encoder_class is None:
        raise ValueError(f"Unsupported provider: {encoder_provider}")
    return encoder_class(name=encoder)
