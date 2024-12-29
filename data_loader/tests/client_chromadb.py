import pytest
import os
import shutil
import tempfile
from pathlib import Path
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Import the function to test
from src.clients.chroma import initialize_vector_store

@pytest.fixture
def temp_db_path():
    """Fixture to provide a temporary database path."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def test_initialize_vector_store_basic(temp_db_path):
    """Test basic initialization of vector store."""
    collection_name = "test_collection"
    vector_store = initialize_vector_store(temp_db_path, collection_name)
    
    assert isinstance(vector_store, ChromaVectorStore)
    assert os.path.exists(temp_db_path)

def test_initialize_vector_store_same_collection(temp_db_path):
    """Test initializing vector store with same collection name twice."""
    collection_name = "test_collection"
    
    # Initialize first time
    vector_store1 = initialize_vector_store(temp_db_path, collection_name)
    
    # Initialize second time with same collection name
    vector_store2 = initialize_vector_store(temp_db_path, collection_name)
    
    assert isinstance(vector_store1, ChromaVectorStore)
    assert isinstance(vector_store2, ChromaVectorStore)
    assert vector_store1._collection.name == vector_store2._collection.name

def test_initialize_vector_store_different_collections(temp_db_path):
    """Test initializing vector store with different collection names."""
    collection1 = "test_collection1"
    collection2 = "test_collection2"
    
    vector_store1 = initialize_vector_store(temp_db_path, collection1)
    vector_store2 = initialize_vector_store(temp_db_path, collection2)
    
    assert vector_store1._collection.name != vector_store2._collection.name

def test_initialize_vector_store_invalid_collection_name(temp_db_path):
    """Test initialization with invalid collection name."""
    with pytest.raises(ValueError):
        initialize_vector_store(temp_db_path, "")  # Empty string collection name

def test_initialize_vector_store_persistence(temp_db_path):
    """Test if data persists between vector store initializations."""
    collection_name = "test_collection"
    
    # Create first instance
    vector_store1 = initialize_vector_store(temp_db_path, collection_name)
    
    # Create second instance with same path and collection
    vector_store2 = initialize_vector_store(temp_db_path, collection_name)
    
    # Verify both instances point to the same collection
    assert vector_store1._collection.name == vector_store2._collection.name
    assert os.path.exists(temp_db_path)

def test_initialize_vector_store_nonexistent_path():
    """Test initialization with a path that doesn't exist."""
    non_existent_path = "/path/that/does/not/exist"
    collection_name = "test_collection"
    
    with pytest.raises(Exception):  # Adjust the exception type based on your implementation
        initialize_vector_store(non_existent_path, collection_name)