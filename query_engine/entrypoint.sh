#!/bin/bash
exec poetry run python handler.py \
    --chroma-db-path "$CHROMA_DB_PATH" \
    --collection-name "$COLLECTION_NAME" \
    --embedding-model "$EMBEDDING_MODEL" \
    --llm-model "$LLM_MODEL"