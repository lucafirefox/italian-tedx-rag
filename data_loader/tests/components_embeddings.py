import pytest
from llama_index.embeddings.openai import OpenAIEmbedding
from src.components.embeddings import initialize_embedding_model

def test_initialize_embedding_model_returns_correct_type():
    """Test that the function returns an OpenAIEmbedding instance"""
    model = initialize_embedding_model("text-embedding-ada-002")
    assert isinstance(model, OpenAIEmbedding)


def test_initialize_embedding_model_type_error():
    """Test that the function raises TypeError for non-string input"""
    with pytest.raises(ValueError):
        initialize_embedding_model(123)