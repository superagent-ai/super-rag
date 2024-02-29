# SuperRag

Super-performant RAG pipeline for AI Agents/Assistants.

## API

### POST /api/v1/ingest

Input example:
```json
{
    "files": [
        {
            "name": "My file",
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
    "chunk_config": {
        "partition_strategy": "auto",
        "split_method": "semantic",
        "min_chunk_tokens": 400,
        "max_token_size": 30,
        "rolling_window_size": 1
    },
    "encoder": {
        "provider": "openai",
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
        "provider": "openai",
        "name": "text-embedding-3-small",
        "dimensions": 384
    },
    "interpreter_mode": False, # Set to True if you wish to run computation Q&A with a code interpreter
    "session_id": "my_session_id" # keeps micro-vm sessions and enables caching 
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
