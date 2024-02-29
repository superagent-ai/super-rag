# SuperRag

Super-performant RAG pipeline for AI Agents/Assistants.

## API

### POST /api/v1/ingest

Input example:
```json
{
    "files": [
        {
            "name": "My file", // Optional
            "url": "https://path-to-my-file.pdf"
        }
    ],
    "document_processor": { // Optional
        "encoder": {
            "dimensions": 384,
            "model_name": "embed-multilingual-light-v3.0",
            "provider": "cohere"
        },
        "unstructured": {
            "hi_res_model_name": "detectron2_onnx",
            "partition_strategy": "auto",
            "process_tables": false
        },
        "splitter": {
            "max_tokens": 400,
            "min_tokens": 30,
            "name": "semantic",
            "prefix_summary": true,
            "prefix_title": true,
            "rolling_window_size": 1
        }
    },
    "vector_database": {
        "type": "qdrant",
        "config": {
            "api_key": "my_api_key",
            "host": "my_qdrant_host"
        }
    },
    "index_name": "my_index",
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
