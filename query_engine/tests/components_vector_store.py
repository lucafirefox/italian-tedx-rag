import pytest
import os
import tempfile
from llama_index.core import VectorStoreIndex
import chromadb

# Import the function to test
from src.components.vector_store import initialize_vector_index

@pytest.fixture
def temp_db_path():
    """Fixture to create a temporary directory for ChromaDB"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def chroma_client(temp_db_path):
    """Fixture to create a ChromaDB client"""
    return chromadb.PersistentClient(path=temp_db_path)


def test_initialize_vector_index_invalid_path():
    """Test initialization with invalid path"""
    # Setup
    invalid_path = "/nonexistent/path/to/nowhere"
    collection_name = "test_collection"
    
    # Execute and Assert
    with pytest.raises(Exception):  # Adjust the exception type based on ChromaDB's specific exception
        initialize_vector_index(collection_name, invalid_path)