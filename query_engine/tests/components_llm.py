import pytest
from llama_index.llms.openai import OpenAI
from pydantic_core import ValidationError
from src.components.llm import initialize_llm

def test_initialize_llm_with_gpt3():
    """Test initialization with GPT-3.5-turbo model"""
    model_name = "gpt-3.5-turbo"
    llm = initialize_llm(model_name)
    
    assert isinstance(llm, OpenAI)
    assert llm.model == model_name

def test_initialize_llm_with_gpt4():
    """Test initialization with GPT-4 model"""
    model_name = "gpt-4"
    llm = initialize_llm(model_name)
    
    assert isinstance(llm, OpenAI)
    assert llm.model == model_name

def test_initialize_llm_empty_string():
    """Test initialization with empty string"""
    model_name = ""
    llm = initialize_llm(model_name)
    
    assert isinstance(llm, OpenAI)
    assert llm.model == ""

def test_initialize_llm_type_error():
    """Test initialization with invalid type"""
    with pytest.raises(ValidationError):
        initialize_llm(123)

def test_initialize_llm_returns_new_instance():
    """Test that each call returns a new instance"""
    model_name = "gpt-3.5-turbo"
    llm1 = initialize_llm(model_name)
    llm2 = initialize_llm(model_name)
    
    assert llm1 is not llm2
    assert isinstance(llm1, OpenAI)
    assert isinstance(llm2, OpenAI)

def test_initialize_llm_with_whitespace():
    """Test initialization with whitespace in model name"""
    model_name = "  gpt-3.5-turbo  "
    llm = initialize_llm(model_name)
    
    assert isinstance(llm, OpenAI)
    assert llm.model == "  gpt-3.5-turbo  "