"""
Main module for initializing and creating a vector index from document transcripts.

This module orchestrates the process of loading documents, setting up necessary components,
and creating a searchable vector index. It handles:
- Document loading from transcripts
- Vector store initialization
- Embedding model setup
- Text splitting configuration
- Language model initialization
- Vector index creation

Dependencies:
    - clients.chroma: Vector store client
    - clients.openai: OpenAI API key validation
    - components.*: Various component initializers
    - llama_index.core: Core indexing functionality
    - loguru: Logging utility
"""

from clients.chroma import initialize_vector_store
from clients.openai import check_openai_api_key
from components.documents import load_documents
from components.embeddings import initialize_embedding_model
from components.llm import initialize_llm
from components.text_splitter import initialize_text_splitter
from components.vector_store import create_vector_index
from llama_index.core import Settings, VectorStoreIndex
from loguru import logger as log


def main(
    transcripts_input_dir: str,
    chroma_db_path: str,
    collection_name: str,
    embedding_model: str,
    llm_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> VectorStoreIndex:
    check_openai_api_key()

    documents = load_documents(input_dir=transcripts_input_dir)

    vector_store = initialize_vector_store(db_path=chroma_db_path, collection_name=collection_name)

    embed_model = initialize_embedding_model(embedding_model)

    text_splitter = initialize_text_splitter(chunk_size, chunk_overlap)

    llm = initialize_llm(llm_model)

    Settings.llm = llm
    Settings.text_splitter = text_splitter
    Settings.embed_model = embed_model
    log.info("Settings initialized")

    vector_index = create_vector_index(documents=documents, vector_store=vector_store, chroma_db_path=chroma_db_path)

    log.info("Indexing complete")

    return vector_index
