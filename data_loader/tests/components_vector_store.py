import pytest
from unittest.mock import Mock, patch
from llama_index.core import Document, StorageContext
from src.components.vector_store import create_vector_index

@pytest.fixture
def sample_documents():
    # Using the id_ property instead of get_doc_id()
    doc1 = Document(text="Test document 1")
    doc2 = Document(text="Test document 2")
    return [doc1, doc2]

@pytest.fixture
def mock_vector_store():
    return Mock()

@pytest.fixture
def mock_storage_context():
    return Mock(spec=StorageContext)

# To suppress the deprecation warning for all tests
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestVectorIndex:
    def test_create_vector_index_successful_creation(self, sample_documents, mock_vector_store):
        """Test successful creation of vector index with valid inputs"""
        with patch('llama_index.core.StorageContext.from_defaults') as mock_storage_context:
            with patch('llama_index.core.VectorStoreIndex.from_documents') as mock_vector_index:
                # Setup
                chroma_db_path = "/tmp/test_db"
                mock_index = Mock()
                mock_vector_index.return_value = mock_index
                
                # Execute
                result = create_vector_index(sample_documents, mock_vector_store, chroma_db_path)
                
                # Assert
                assert result == mock_index
                mock_storage_context.assert_called_once_with(vector_store=mock_vector_store)
                mock_vector_index.assert_called_once()
                mock_index.storage_context.persist.assert_called_once_with(persist_dir=chroma_db_path)

    def test_create_vector_index_empty_documents(self, mock_vector_store):
        """Test vector index creation with empty document list"""
        with patch('llama_index.core.StorageContext.from_defaults') as mock_storage_context:
            with patch('llama_index.core.VectorStoreIndex.from_documents') as mock_vector_index:
                # Setup
                empty_documents = []
                chroma_db_path = "/tmp/test_db"
                
                # Execute
                result = create_vector_index(empty_documents, mock_vector_store, chroma_db_path)
                
                # Assert
                mock_storage_context.assert_called_once()
                mock_vector_index.assert_called_once()

    @pytest.mark.parametrize("invalid_path", [
        "",
        None,
        123  # non-string path
    ])
    def test_create_vector_index_invalid_path(self, sample_documents, mock_vector_store, invalid_path):
        """Test vector index creation with invalid chroma_db_path"""
        with pytest.raises((ValueError, TypeError)):
            create_vector_index(sample_documents, mock_vector_store, invalid_path)

    def test_create_vector_index_storage_context_error(self, sample_documents, mock_vector_store):
        """Test handling of StorageContext creation error"""
        with patch('llama_index.core.StorageContext.from_defaults', side_effect=Exception("Storage context error")):
            with pytest.raises(Exception) as exc_info:
                create_vector_index(sample_documents, mock_vector_store, "/tmp/test_db")
            assert "Storage context error" in str(exc_info.value)

    def test_create_vector_index_persistence_error(self, sample_documents, mock_vector_store):
        """Test handling of persistence error"""
        with patch('llama_index.core.StorageContext.from_defaults') as mock_storage_context:
            with patch('llama_index.core.VectorStoreIndex.from_documents') as mock_vector_index:
                # Setup
                mock_index = Mock()
                mock_index.storage_context.persist.side_effect = Exception("Persistence error")
                mock_vector_index.return_value = mock_index
                
                # Execute and Assert
                with pytest.raises(Exception) as exc_info:
                    create_vector_index(sample_documents, mock_vector_store, "/tmp/test_db")
                assert "Persistence error" in str(exc_info.value)

    def test_create_vector_index_vector_store_none(self):
        """Test vector index creation with None vector_store"""
        with pytest.raises(ValueError):
            create_vector_index([], None, "/tmp/test_db")