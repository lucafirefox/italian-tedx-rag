from llama_index.core import StorageContext, VectorStoreIndex
from loguru import logger as log


def create_vector_index(documents: list, vector_store: VectorStoreIndex, chroma_db_path: str) -> VectorStoreIndex:
    """
    Create and persist a vector store index from a list of documents.

    Args:
        documents (list): List of documents to be indexed in the vector store
        vector_store (VectorStoreIndex): The vector store instance to use for indexing
        chroma_db_path (str): File system path where the vector store index will be persisted

    Returns:
        VectorStoreIndex: An initialized vector store index containing the embedded documents
    """

    log.info("Creating vector store index from documents.")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_index = VectorStoreIndex.from_documents(documents, storage_context)

    log.info("Persisting vector store index to database.")
    vector_index.storage_context.persist(persist_dir=chroma_db_path)

    return vector_index
