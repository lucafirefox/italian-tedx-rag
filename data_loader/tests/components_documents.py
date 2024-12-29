import pytest
from pathlib import Path
from llama_index.core import Document
from src.components.documents import extract_metadata, load_documents 

# Test data for extract_metadata
VALID_FILENAMES = [
    (
        "Important Speech｜John Doe｜New York[abc123]",
        {
            "speech_name": "Important Speech",
            "speech_author": "John Doe",
            "speech_location": "New York",
            "youtube_id": "abc123"
        }
    ),
    (
        "path/to/Keynote Address｜Jane Smith｜London Conference[xyz789]",
        {
            "speech_name": "Keynote Address",
            "speech_author": "Jane Smith",
            "speech_location": "London Conference",
            "youtube_id": "xyz789"
        }
    ),
]

INVALID_FILENAMES = [
    "Invalid_filename.txt",
    "No_Separators_Here",
    "Partial｜Separator",
    "Missing[Brackets]",
]

# Tests for extract_metadata function
@pytest.mark.parametrize("filename, expected", VALID_FILENAMES)
def test_extract_metadata_valid(filename, expected):
    """Test extract_metadata with valid filenames"""
    result = extract_metadata(filename)
    assert result == expected

@pytest.mark.parametrize("filename", INVALID_FILENAMES)
def test_extract_metadata_invalid(filename):
    """Test extract_metadata with invalid filenames"""
    result = extract_metadata(filename)
    assert result is None

def test_extract_metadata_empty():
    """Test extract_metadata with empty string"""
    result = extract_metadata("")
    assert result is None

# Tests for load_documents function
def test_load_documents(tmp_path):
    """Test load_documents with temporary directory"""
    # Create temporary test files
    test_file = tmp_path / "Test Speech｜Author Name｜Test Location[test123]"
    test_file.write_text("Test content")

    # Load documents
    docs = load_documents(str(tmp_path))

    # Assertions
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].metadata["speech_name"] == "Test Speech"
    assert docs[0].metadata["speech_author"] == "Author Name"
    assert docs[0].metadata["speech_location"] == "Test Location"
    assert docs[0].metadata["youtube_id"] == "test123"

def test_load_documents_empty_dir(tmp_path):
    """Test load_documents with empty directory"""
    # Since SimpleDirectoryReader raises ValueError for empty directories
    with pytest.raises(ValueError, match="No files found in"):
        docs = load_documents(str(tmp_path))

@pytest.mark.parametrize("filename", INVALID_FILENAMES)
def test_load_documents_invalid_filenames(tmp_path, filename):
    """Test load_documents with files having invalid filenames"""
    # Create test file with invalid filename
    test_file = tmp_path / filename
    test_file.write_text("Test content")

    # Load documents
    docs = load_documents(str(tmp_path))

    # Verify document is loaded but metadata is None
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].metadata is None or "speech_name" not in docs[0].metadata

def test_load_documents_multiple_files(tmp_path):
    """Test load_documents with multiple files"""
    # Create multiple test files
    files = [
        ("Speech 1｜Author 1｜Location 1[id1]", "Content 1"),
        ("Speech 2｜Author 2｜Location 2[id2]", "Content 2"),
    ]
    
    for filename, content in files:
        (tmp_path / filename).write_text(content)

    # Load documents
    docs = load_documents(str(tmp_path))

    # Assertions
    assert len(docs) == 2
    for i, doc in enumerate(docs):
        assert isinstance(doc, Document)
        assert doc.metadata["speech_name"] == f"Speech {i+1}"
        assert doc.metadata["speech_author"] == f"Author {i+1}"
        assert doc.metadata["speech_location"] == f"Location {i+1}"
        assert doc.metadata["youtube_id"] == f"id{i+1}"

def test_load_documents_nonexistent_dir():
    """Test load_documents with non-existent directory"""
    with pytest.raises(Exception):  # Adjust exception type as needed
        load_documents("nonexistent_directory")