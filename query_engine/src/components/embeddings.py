from llama_index.embeddings.openai import OpenAIEmbedding


def initialize_embedding_model(model_name: str) -> OpenAIEmbedding:
    """
    Initialize an OpenAI embedding model with the specified model name.

    Args:
        model_name (str): The name of the OpenAI embedding model to initialize
                         (e.g., 'text-embedding-ada-002')

    Returns:
        OpenAIEmbedding: An initialized OpenAI embedding model instance for use with LlamaIndex
                        document embeddings and queries
    """
    return OpenAIEmbedding(model=model_name)
