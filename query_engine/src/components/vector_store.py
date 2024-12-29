__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore


def initialize_vector_index(collection_name: str, chroma_db_path: str) -> VectorStoreIndex:
    """
    Initialize and return a VectorStoreIndex using ChromaDB as the vector store.

    This function creates a persistent ChromaDB client, retrieves a collection,
    and initializes a vector store index for document storage and retrieval.

    Args:
        collection_name (str): Name of the ChromaDB collection to use or create
        chroma_db_path (str): File system path where ChromaDB will persist its data

    Returns:
        VectorStoreIndex: An initialized vector store index ready for document operations
    """

    # Initialize ChromaDB client with persistent storage
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    chroma_collection = chroma_client.get_collection(name=collection_name)

    # Create vector store and storage context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=chroma_db_path)

    # Initialize and return the vector store index
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )
