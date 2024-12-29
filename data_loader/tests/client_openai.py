import os
import pytest
from unittest.mock import patch
from src.clients.openai import check_openai_api_key

def test_check_openai_api_key_when_key_exists():
    """Test when OPENAI_API_KEY environment variable is set"""
    test_api_key = "test-api-key-123"
    
    with patch.dict(os.environ, {'OPENAI_API_KEY': test_api_key}):
        result = check_openai_api_key()
        assert result == test_api_key

def test_check_openai_api_key_when_key_missing():
    """Test when OPENAI_API_KEY environment variable is not set"""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(OSError) as exc_info:
            check_openai_api_key()
        assert str(exc_info.value) == "OPENAI_API_KEY environment variable not set"

def test_check_openai_api_key_when_key_empty():
    """Test when OPENAI_API_KEY environment variable is empty"""
    with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
        with pytest.raises(OSError) as exc_info:
            check_openai_api_key()
        assert str(exc_info.value) == "OPENAI_API_KEY environment variable not set"
