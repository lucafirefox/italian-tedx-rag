FROM python:3.12.3-slim-bullseye

LABEL maintainer="Luca Ferrario <lucaferrario199@gmail.com>"
LABEL description="Data loader for TEDx Italian Speeches RAG"
LABEL version="0.1.0"

# Install system dependencies and clean up in a single layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=1.8.5
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Set the working directory for the application
WORKDIR /app

# Copy dependency files, source code and entrypoint script
COPY pyproject.toml poetry.lock ./
COPY src/. .
COPY entrypoint.sh ./

RUN chmod +x entrypoint.sh

# Install dependencies using Poetry
RUN poetry install --no-interaction --no-ansi --no-root

# Set environment variables:
# PYTHONUNBUFFERED=1: Ensures Python output is sent straight to terminal (helps with logging)
# PYTHONDONTWRITEBYTECODE=1: Prevents Python from writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 

ENTRYPOINT ["./entrypoint.sh"]