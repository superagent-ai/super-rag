# SuperRag

Super-performant RAG pipeline for AI Agents/Assistants.

## API

### POST /api/v1/ingest

Input example:
```json
{
    "files": [
        {
            "type": "PDF",
            "url": "https://path-to-my-file.pdf"
        }
    ],
    "vector_database": {
        "type": "qdrant",
        "config": {
            "api_key": "my_api_key",
            "host": "my_qdrant_host"
        }
    },
    "index_name": "my_index",
    "encoder": {
        "type": "openai",
        "name": "text-embedding-3-small",
        "dimensions": 1536  # encoder depends on the provider and model
    },
    "webhook_url": "https://my-webhook-url"
}
```

### POST /api/v1/query

Input example:
```json
{
    "input": "A query",
    "vector_database": {
        "type": "qdrant",
        "config": {
            "api_key": "my_api_key",
            "host": "my_qdrant_host"
        }
    },
    "index_name": "my_index",
    "encoder": {
        "type": "openai",
        "name": "text-embedding-3-small",
    } 
}
```

### DELETE /api/v1/delete

Input example:
```json
{
    "file_url": "A file url to delete",
    "vector_database": {
        "type": "qdrant",
        "config": {
            "api_key": "my_api_key",
            "host": "my_qdrant_host"
        }
    },
    "index_name": "my_index",
}
```

## Supported file types

- PDF
- TXT
- MARKDOWN
- PPTX
- DOCX

## Supported vector databases

- qdrant
- pinecone
- weaviate
- astra
