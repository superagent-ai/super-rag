<div align="center">
	<img width="100px" src="https://github.com/homanp/superagent/assets/2464556/eb51fa38-4a2a-4c41-b348-d3c1abc04234" />
	<h1>Super-Rag</h1>
	<p>
		<b>Super-performant RAG pipeline for AI apps.</b>
	</p>
	<br>
    <p align="center">
        <a href="#-key-features">Features</a> •
        <a href="#-installation">Installation</a> •
        <a href="#-how-to-use">How to use</a> •
        <a href="#-cloud-api">Cloud API</a>
    </p>
</div>


## ✅ Key features
 - Supports multiple document formats and vector databases.
 - Provides a production ready REST API.
 - Customizable splitting/chunking.
 - Includes options for encoding data using different encoding models both propriatory and open source.
 - Built in code interpreter mode for computational question & answer scenarios.
 - Allows session management through unique IDs for caching purposes.

## ☁️ Cloud API 

Easiest way to get started is to use our [Cloud API](https://d3jvqvcd9u4534.cloudfront.net). This API is free to use (within reasonable limits).

## 📦  Installation

1. Clone the repository
    ```bash
    git clone https://github.com/superagent-ai/super-rag 
    cd super-rag 
    ```

2. Setup virtual environment
    ```bash
    # Using virtualenv 
    virtualenv env 
    source env/bin/activate 
    
    # Or using venv 
    python3 -m venv env 
    source env/bin/activate 
    ```

3. Install requried packages
    ```bash
    poetry install
    ```

4. Rename `.env.example` to `.env` and set your environment variables

5. Run server
    ```bash
    uvicorn main:app --reload
    ```
## 🤖 Interpreter mode
Super-Rag has built in support for running computational Q&A using code interpreters powered by [E2B.dev](https://e2b.dev) custom runtimes. You can signup to receive an API key to leverage they sandboxes in a cloud environment or setup your own by following [these instructions](https://github.com/e2b-dev/infra). 

## 🚀 How to use 
Super-Rag comes with a built in REST API powered by FastApi. 

### Ingest documents
```json
// POST: /api/v1/ingest

// Payload
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
            "api_key": "YOUR API KEY",
            "host": "THE QDRANT HOST"
        }
    },
    "index_name": "my_index",
    "webhook_url": "https://my-webhook-url"
}
```

### Query documents
```json
// POST: /api/v1/query

// Payload
{
    "input": "What is ReAct",
    "vector_database": {
            "type": "qdrant",
            "config": {
            "api_key": "YOUR API KEY",
            "host": "THE QDRANT HOST"
        }
        },
    "index_name": "YOUR INDEX",
    "interpreter_mode": true,
    "encoder": {
        "provider": "cohere",
        "name": "embed-multilingual-light-v3.0",
        "dimensions": 384
    },
    "exclude_fields": ["metadata"], // Exclude specific fields
    "interpreter_mode": False, // Set to True if you wish to run computation Q&A with a code interpreter
    "session_id": "my_session_id" // keeps micro-vm sessions and enables caching 
}
```

### Delete document
```json
// POST: /api/v1/delete

// Payload
{
    "file_url": "A file url to delete",
    "vector_database": {
        "type": "qdrant",
        "config": {
            "api_key": "YOUR API KEY",
            "host": "THE QDRANT HOST"
        }
    },
    "index_name": "my_index",
}

```

## 🧠 Supportd encoders
- OpenAi
- Cohere
- HuggingFace
- FastEmbed
- Mistral (coming soon)
- Anthropic (coming soon)


## 🗃 Supported vector databases
- Pinecone
- Qdrant
- Weaviate
- Astra
- Supabase (coming soon)
- Chroma (coming soon)
