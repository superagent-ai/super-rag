from typing import List
from fastapi import UploadFile
from tempfile import NamedTemporaryFile
from llama_index import Document, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser


class EmbeddingService:
    def __init__(self, files: List[UploadFile]):
        self.files = files

    async def generate_chunks(self):
        documents = []
        for file in self.files:
            with NamedTemporaryFile(
                suffix=file.filename.split(".")[-1], delete=True
            ) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()
                reader = SimpleDirectoryReader(input_files=[temp_file.name])
                docs = reader.load_data()
                documents.append(docs)
        return documents
