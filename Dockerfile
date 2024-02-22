# Use an official Python runtime as a parent image
FROM python:3.11 AS builder

# Copy the current directory contents into the container
COPY . .

# Install Poetry
RUN pip install poetry

# Copy only dependency files for layer caching
COPY pyproject.toml poetry.lock ./

# Install the required packages of the application
RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# Make port 80 available to the world outside this container
ENV PORT="8080"

# Run main.py when the container launches
CMD exec gunicorn --bind :$PORT --workers 2 --timeout 0  --worker-class uvicorn.workers.UvicornWorker  --threads 8 main:app