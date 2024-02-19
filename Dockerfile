# Use the python latest image | slim saves ~400MB compared to standard
FROM python:3.11-slim


RUN pip install -U pip \
    pip install poetry;

WORKDIR /app

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache
# Copy only dependency files for layer caching
COPY pyproject.toml ./

# Install the required packages of the application into .venv
RUN poetry run pip install -f https://download.pytorch.org/whl/cpu/torch_stable.html torch[cpu] torchvision[cpu]
#
# RUN poetry fastapi uvicorn weaviate-client llama-index pinecone-client qdrant-client ruff black flake8 vulture python-decouple semantic-router astrapy openai tqdm cohere cmake pypdf docx2txt python-dotenv unstructured 
RUN poetry install 

ENV PATH="/app/.venv/bin:$PATH"

# Make port 8080 available to the world outside this container
ENV PORT="8080"

COPY . ./
# Run app.py when the container launches
