from llama_index.core.text_splitter import TokenTextSplitter
from loguru import logger as log


def initialize_text_splitter(chunk_size: int, chunk_overlap: int) -> TokenTextSplitter:
    """
    Initialize a TokenTextSplitter for document chunking in LlamaIndex.

    Args:
        chunk_size (int, optional): The size of each text chunk in tokens.
        chunk_overlap (int, optional): The number of overlapping tokens between chunks.

    Returns:
        TokenTextSplitter: A configured text splitter instance for document processing
    """
    log.info("Initializing TokenTextSplitter. Chunk size: {}, Chunk overlap: {}", chunk_size, chunk_overlap)
    return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
