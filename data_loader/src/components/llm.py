from llama_index.llms.openai import OpenAI
from loguru import logger as log


def initialize_llm(model_name: str) -> OpenAI:
    """
    Initialize the OpenAI language model for use with LlamaIndex.

    Args:
        model_name (str): The name of the OpenAI model to initialize (e.g., 'gpt-3.5-turbo', 'gpt-4')

    Returns:
        OpenAI: An initialized OpenAI language model instance configured for LlamaIndex operations
    """
    log.info("Initializing OpenAI language model: {}", model_name)
    return OpenAI(model=model_name)
