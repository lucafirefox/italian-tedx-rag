"""
Main entry point for the RAG (Retrieval-Augmented Generation) system.

This module initializes and launches a RAG system with the following components:
- Embedding model for text vectorization
- Language Model (LLM) for text generation
- Vector store for document storage and retrieval
- Gradio interface for user interaction

The system allows users to query a document collection using natural language,
retrieving relevant context and generating appropriate responses.

Dependencies:
    - llama_index
    - gradio
    - chromadb
    - loguru
"""

from components.embeddings import initialize_embedding_model
from clients.openai import check_openai_api_key
from components.llm import initialize_llm
from components.vector_store import initialize_vector_index
from interfaces import GradioInterface, RAGQueryInterface
from llama_index.core import Settings
from loguru import logger as log


def main(chroma_db_path: str, collection_name: str, embedding_model: str, llm_model: str):
    
    check_openai_api_key()
    
    # Initialize LLM and embedding model
    llm = initialize_llm(llm_model)
    embed_model = initialize_embedding_model(embedding_model)

    Settings.llm = llm
    Settings.embed_model = embed_model
    log.info("Settings initialized")

    # Initialize vector index
    index = initialize_vector_index(
        collection_name=collection_name,
        chroma_db_path=chroma_db_path,
    )

    # Initialize RAG query interface and Gradio interface
    rag_interface = RAGQueryInterface(index)
    gradio_interface = GradioInterface(rag_interface)

    # Launch the Gradio interface
    gradio_interface.launch(share=False, server_name="0.0.0.0", server_port=8000)
