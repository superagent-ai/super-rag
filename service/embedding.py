import asyncio
import copy
from tempfile import NamedTemporaryFile
from typing import Any, List, Optional

import numpy as np
import requests
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from tqdm import tqdm

import encoders
from encoders import BaseEncoder
from models.file import File
from models.ingest import EncoderEnum
from models.document import Document
from service.vector_database import get_vector_service
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
        }
        try:
            return suffixes[type]
        except KeyError:
            raise ValueError("Unsupported datasource type")

    async def generate_documents(self) -> List[Document]:
        documents = []
        for file in tqdm(self.files, desc="Generating documents"):
            suffix = self._get_datasource_suffix(file.type.value)
            with NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
                with requests.get(url=file.url) as response:  # Add context manager here
                    temp_file.write(response.content)
                    temp_file.flush()
                elements = partition(file=temp_file, include_page_breaks=True)
                chunks = chunk_by_title(elements)
                for chunk in chunks:
                    documents.append(
                        Document(
                            id=file.url,
                            text=chunk.text,
                            file_url=file.url,
                            metadata={**chunk.metadata.to_dict()},
                        )
                    )
        return documents

    async def generate_embeddings(
        self,
        documents: List[Document],
        encoder: BaseEncoder,
        index_name: Optional[str] = None,
    ) -> List[tuple[str, list, dict[str, Any]]]:
        pbar = tqdm(total=len(documents), desc="Generating embeddings")

        async def generate_embedding(document: Document):
            if document is not None:
                embeddings: List[np.ndarray] = [
                    np.array(e) for e in encoder([document.text])
                ]
                embedding = (
                    document.id,
                    embeddings[0].tolist(),
                    {
                        **document.metadata,
                        "content": document.text,
                    },
                )
                pbar.update()
                return embedding

        tasks = [generate_embedding(document) for document in documents]
        embeddings = await asyncio.gather(*tasks)
        pbar.close()
        vector_service = get_vector_service(
            index_name=index_name or self.index_name,
            credentials=self.vector_credentials,
            encoder=encoder,
        )
        await vector_service.upsert(embeddings=[e for e in embeddings if e is not None])

        return [e for e in embeddings if e is not None]

    async def generate_summary_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        pbar = tqdm(total=len(documents), desc="Summarizing documents")
        pages = {}
        for document in documents:
            page_number = document.metadata.get("page_number")
            if page_number not in pages:
                doc = copy.deepcopy(document)
                doc.text = await completion(document=doc)
                pages[page_number] = doc
            else:
                pages[page_number].text += document.text
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
