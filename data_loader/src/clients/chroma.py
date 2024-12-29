__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger as log


def initialize_vector_store(db_path: str, collection_name: str) -> ChromaVectorStore:
    """
    Initialize and return a ChromaVectorStore instance.

    Args:
        db_path (str): Path to the ChromaDB database
        collection_name (str): Name of the collection to create or get

    Returns:
        ChromaVectorStore: Initialized vector store instance
    """

    # Initialize the persistent client
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Get or create the collection
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

    # Initialize the vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    log.info("Initialized ChromaVectorStore. Collection: {}, DB path: {}", collection_name, db_path)

    return vector_store
