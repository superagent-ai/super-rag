import requests

from typing import List
from fastapi import UploadFile
from tempfile import NamedTemporaryFile
from llama_index import Document, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from models.file import File


class EmbeddingService:
    def __init__(self, files: List[File], index_name: str):
        self.files = files
        self.index_name = index_name

    def _get_datasource_suffix(self, type: str) -> str:
        suffixes = {"TXT": ".txt", "PDF": ".pdf", "MARKDOWN": ".md"}
        try:
            return suffixes[type]
        except KeyError:
            raise ValueError("Unsupported datasource type")

    async def generate_documents(self):
        documents = []
        for file in self.files:
            print(file.type.value)
            suffix = self._get_datasource_suffix(file.type.value)
            with NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
                response = requests.get(url=file.url)
                temp_file.write(response.content)
                temp_file.flush()
                reader = SimpleDirectoryReader(input_files=[temp_file.name])
                docs = reader.load_data()
                documents.append(docs)
        return documents
