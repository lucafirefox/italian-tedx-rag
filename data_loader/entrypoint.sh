#!/bin/bash
exec poetry run python handler.py \
    --transcripts-input-dir "$TRANSCRIPTS_INPUT_DIR" \
    --chroma-db-path "$CHROMA_DB_PATH" \
    --collection-name "$COLLECTION_NAME" \
    --embedding-model "$EMBEDDING_MODEL" \
    --llm-model "$LLM_MODEL" \
    --chunk-size "$CHUNK_SIZE" \
    --chunk-overlap "$CHUNK_OVERLAP"