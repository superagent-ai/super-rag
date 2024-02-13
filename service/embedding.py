import asyncio
import copy
import uuid
from tempfile import NamedTemporaryFile
from typing import Any, List, Optional

import numpy as np
import requests
from tqdm import tqdm
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition

import encoders
from encoders import BaseEncoder
from models.document import BaseDocument, BaseDocumentChunk
from models.file import File
from models.ingest import EncoderEnum
from service.vector_database import get_vector_service
from utils.logger import logger
from utils.summarise import completion


class EmbeddingService:
    def __init__(self, files: List[File], index_name: str, vector_credentials: dict):
        self.files = files
        self.index_name = index_name
        self.vector_credentials = vector_credentials

    def _get_datasource_suffix(self, type: str) -> str:
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
        with NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
            with requests.get(url=file.url) as response:
                temp_file.write(response.content)
                temp_file.flush()
            elements = partition(
                file=temp_file, include_page_breaks=True, strategy=strategy
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
                    elements, max_characters=500, combine_text_under_n_chars=0
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

    async def generate_embeddings(
        self,
        documents: List[BaseDocumentChunk],
        encoder: BaseEncoder,
        index_name: Optional[str] = None,
    ) -> List[BaseDocumentChunk]:
        pbar = tqdm(total=len(documents), desc="Generating embeddings")

        async def safe_generate_embedding(
            chunk: BaseDocumentChunk,
        ) -> BaseDocumentChunk | None:
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

                logger.info(f"Embedding: {chunk.id}, metadata: {chunk.metadata}")
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
        )
        try:
            await vector_service.upsert(chunks=chunks_with_embeddings)
        except Exception as e:
            logger.error(f"Error upserting embeddings: {e}")
            raise Exception(f"Error upserting embeddings: {e}")

        return chunks_with_embeddings

    # TODO: Do we summarize the documents or chunks here?
    async def generate_summary_documents(
        self, documents: List[BaseDocumentChunk]
    ) -> List[BaseDocumentChunk]:
        pbar = tqdm(total=len(documents), desc="Summarizing documents")
        pages = {}
        for document in documents:
            page_number = document.page_number
            if page_number not in pages:
                doc = copy.deepcopy(document)
                doc.content = await completion(document=doc)
                pages[page_number] = doc
            else:
                pages[page_number].content += document.content
            pbar.update()
        pbar.close()
        summary_documents = list(pages.values())
        return summary_documents


def get_encoder(*, encoder_type: EncoderEnum) -> encoders.BaseEncoder:
    encoder_mapping = {
        EncoderEnum.cohere: encoders.CohereEncoder,
        EncoderEnum.openai: encoders.OpenAIEncoder,
        EncoderEnum.huggingface: encoders.HuggingFaceEncoder,
    }

    encoder_class = encoder_mapping.get(encoder_type)
    if encoder_class is None:
        raise ValueError(f"Unsupported encoder: {encoder_type}")
    return encoder_class()
