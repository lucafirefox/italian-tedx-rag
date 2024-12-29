import pytest
from llama_index.core.text_splitter import TokenTextSplitter
from src.components.text_splitter import initialize_text_splitter


def test_initialize_text_splitter_valid_parameters():
    """Test initialization with valid parameters."""
    chunk_size = 1000
    chunk_overlap = 200
    
    splitter = initialize_text_splitter(chunk_size, chunk_overlap)
    
    assert isinstance(splitter, TokenTextSplitter)
    assert splitter.chunk_size == chunk_size
    assert splitter.chunk_overlap == chunk_overlap

def test_initialize_text_splitter_zero_overlap():
    """Test initialization with zero chunk overlap."""
    chunk_size = 500
    chunk_overlap = 0
    
    splitter = initialize_text_splitter(chunk_size, chunk_overlap)
    
    assert isinstance(splitter, TokenTextSplitter)
    assert splitter.chunk_size == chunk_size
    assert splitter.chunk_overlap == chunk_overlap

def test_initialize_text_splitter_equal_size_and_overlap():
    """Test initialization when chunk size equals overlap."""
    chunk_size = 300
    chunk_overlap = 300
    
    splitter = initialize_text_splitter(chunk_size, chunk_overlap)
    
    assert isinstance(splitter, TokenTextSplitter)
    assert splitter.chunk_size == chunk_size
    assert splitter.chunk_overlap == chunk_overlap

def test_initialize_text_splitter_negative_values():
    """Test initialization with negative values should raise ValueError."""
    with pytest.raises(ValueError):
        initialize_text_splitter(-100, 50)
    
    with pytest.raises(ValueError):
        initialize_text_splitter(100, -50)

def test_initialize_text_splitter_overlap_larger_than_chunk():
    """Test initialization when overlap is larger than chunk size should raise ValueError."""
    with pytest.raises(ValueError):
        initialize_text_splitter(100, 200)

def test_initialize_text_splitter_zero_chunk_size():
    """Test initialization with zero chunk size should raise ValueError."""
    with pytest.raises(ValueError):
        initialize_text_splitter(0, 0)

@pytest.mark.parametrize("chunk_size,chunk_overlap", [
    (1000, 100),
    (500, 50),
    (2000, 400),
])
def test_initialize_text_splitter_multiple_values(chunk_size, chunk_overlap):
    """Test initialization with multiple different valid parameter combinations."""
    splitter = initialize_text_splitter(chunk_size, chunk_overlap)
    
    assert isinstance(splitter, TokenTextSplitter)
    assert splitter.chunk_size == chunk_size
    assert splitter.chunk_overlap == chunk_overlap
